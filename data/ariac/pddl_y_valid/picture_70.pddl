(define (problem picture_70)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery blue_pump red_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_regulator)
        (clear blue_battery)
        (part_at blue_pump table)
        (on blue_battery blue_pump)
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
