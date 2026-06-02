(define (problem picture_72)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery blue_pump red_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_pump)
        (clear blue_battery)
        (clear red_regulator)
        (part_at blue_pump table)
        (part_at blue_battery table)
        (part_at red_regulator table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at red_regulator regulator_placement)
            (part_at blue_battery battery_placement)
        )
    )
)
