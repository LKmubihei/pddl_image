(define (problem picture_63)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery blue_pump red_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_battery)
        (clear red_regulator)
        (part_at red_battery table)
        (part_at blue_pump table)
        (on red_regulator blue_pump)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at red_regulator regulator_placement)
            (part_at red_battery battery_placement)
        )
    )
)
